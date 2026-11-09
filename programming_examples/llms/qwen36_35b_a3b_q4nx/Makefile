# Copyright (C) 2026, Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Qwen3.6-35B-A3B Q4NX decode on NPU2 (MLIR-AIR), FastFlowLM's layout: the 40
# layers in one dispatch and the Q8 LM head in a second, the router choosing
# and fetching the experts on device. Weights come from FastFlowLM's model.q4nx
# bundle (FastFlowLM/Qwen3.6-35B-A3B-NPU2; QWEN36_FLM_DIR names a local copy).
#
#   make compile   Build the decode and LM head ELFs
#   make run       Greedy decode of PROMPT, checked against the CPU reference
#   make verify    Each stage alone, the decode, and the reference against HF
#                  transformers' Qwen3.5-MoE layers
#   make profile   Time per token at context CTX, no reference
#   make kernels   Every kernel test

srcdir := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
PYTHON ?= python3
DECODE := $(PYTHON) $(srcdir)/decode.py

MAXL      ?= 64
GEN       ?= 8
PROMPT    ?= The capital of France
CTX       ?= 1024
BENCH     ?= 20
BUILD_DIR ?= build_decode
STAGES    := moe_block gdn_layer attn_layer lm_head
KERNELS   := q4k_mvm router gdn_prep gdn_decode gate rope attn ffn

.PHONY: compile run verify profile kernels clean

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
	$(MAKE) -f $(srcdir)/Makefile run
	$(PYTHON) $(srcdir)/check_vs_hf.py

profile:
	mkdir -p build_ctx$(CTX) && cd build_ctx$(CTX) && \
		$(DECODE) --maxl $(CTX) --gen 2 --no-ref --bench $(BENCH) --bench-l $(CTX)

kernels:
	mkdir -p build_kernels && cd build_kernels && \
		for k in $(KERNELS); do \
			$(PYTHON) $(srcdir)/kernel_tests/$$k.py || exit 1; done

clean:
	rm -rf $(BUILD_DIR) build_* *.o
