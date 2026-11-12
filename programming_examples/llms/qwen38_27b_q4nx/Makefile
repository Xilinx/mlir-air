# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3.8-27B Q4NX decode on NPU2 (MLIR-AIR), FastFlowLM's layout: the 64
# layers in one dispatch and the Q8 LM head in a second. Weights come from
# FastFlowLM's model.q4nx bundle (FastFlowLM/Qwen3.8-27B-NPU2; QWEN38_FLM_DIR
# names a local copy).
#
#   make compile         Build the decode and LM head ELFs
#   make run             Greedy decode of PROMPT, checked against the CPU reference
#   make verify          Each stage alone, the decode, and the reference against
#                        HF transformers' Qwen3.5 layers
#   make profile         TTFT and decode throughput of PROMPT, no reference
#   make profile-decode  Decode tok/s at each of CONTEXTS (one build each)
#   make kernels         Every kernel test
#
# The prompt goes through the decode a token at a time.

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3
DECODE := $(PYTHON) $(srcdir)/decode.py
comma := ,

MAXL      ?= 64
GEN       ?= 8
PROMPT    ?= The capital of France
N_TOKENS  ?= 32
CTX       ?= 1024
CONTEXTS  ?= 1024,4096,8192
BENCH     ?= 20
BUILD_DIR ?= build_decode
STAGES    := mlp_block gdn_layer attn_layer lm_head
KERNELS   := q4k_mvm gdn_prep gdn_decode gate rope attn

.PHONY: compile run verify profile profile-decode kernels clean

compile:
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		$(DECODE) --compile-only --maxl $(MAXL)

run:
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		$(DECODE) --maxl $(MAXL) --gen $(GEN) --prompt "$(PROMPT)"

verify:
	for s in $(STAGES); do \
		mkdir -p build_$$s && (cd build_$$s && $(PYTHON) $(srcdir)/$$s.py) \
			|| exit 1; done
	$(MAKE) -f $(srcdir)/Makefile run GEN=4
	$(PYTHON) $(srcdir)/check_vs_hf.py

# The context the build holds covers the prompt and the generated tokens.
profile:
	mkdir -p $(BUILD_DIR) && cd $(BUILD_DIR) && \
		$(DECODE) --maxl $$(( ($(N_TOKENS) + 64 + 15) / 16 * 16 )) \
			--gen $(N_TOKENS) --no-ref --prompt "$(PROMPT)"

# The context is compiled in: one build per point, each at its own maxl.
profile-decode:
	for c in $(subst $(comma), ,$(CONTEXTS)); do \
		mkdir -p build_ctx$$c && (cd build_ctx$$c && \
			$(DECODE) --maxl $$c --gen 2 --no-ref --bench $(BENCH) --bench-l $$c) \
			|| exit 1; done

kernels:
	mkdir -p build_kernels && cd build_kernels && \
		for k in $(KERNELS); do \
			$(PYTHON) $(srcdir)/kernel_tests/$$k.py || exit 1; done

clean:
	rm -rf $(BUILD_DIR) build_* *.o
